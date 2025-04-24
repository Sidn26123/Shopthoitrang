package ptithcm.entity;

import java.math.BigDecimal;
import java.util.Collection;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.FetchType;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;
import javax.persistence.OneToMany;
import javax.persistence.Table;

@Entity
@Table(name="LOAISANPHAM")
public class LoaiSanPhamEntity{
	@Id
	@Column(name="MALOAI")
	@GeneratedValue
	private String maLoai;
	@Column(name="TENLOAI")
	private String tenLoai;
	
	@OneToMany(mappedBy = "loai", fetch = FetchType.LAZY )
	private Collection<KieuSanPhamEntity> kieus;

	public String getMaLoai() {
		return maLoai;
	}

	public void setMaLoai(String maLoai) {
		this.maLoai = maLoai;
	}

	public String getTenLoai() {
		return tenLoai;
	}

	public void setTenLoai(String tenLoai) {
		this.tenLoai = tenLoai;
	}

	public Collection<KieuSanPhamEntity> getKieus() {
		return kieus;
	}

	public void setKieus(Collection<KieuSanPhamEntity> kieus) {
		this.kieus = kieus;
	}


	
}