package ptithcm.dao;

import java.util.List;

import ptithcm.entity.NguoiDungEntity;

public interface nguoiDungDao{
	
	public void addUser(NguoiDungEntity user);
	public void updateUser(NguoiDungEntity user);
	public List<NguoiDungEntity>  getAllUserByRole(Integer maQuyen);
	public NguoiDungEntity findUserById(Integer maNd);
	public NguoiDungEntity findUserByNameAndEmail(String userName, String email);
	
	
}